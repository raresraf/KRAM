import './App.scss'

import * as React from 'react'

import Main from './components/Main'

export type Page = 'main'

export default function App() {
  const [page] = React.useState<Page>('main')

  const content = page === 'main' ? <Main />: null

  return (
    <div className='App'>
      {content}
    </div>
  )
}
